from flask import Flask, render_template, request, Response
from wtforms import Form, FloatField, IntegerField, BooleanField, StringField, validators
import time, sys

from . import app

class AppForm(Form):
  pi = IntegerField('pi', [validators.DataRequired()])
  m = FloatField('m', [validators.DataRequired()])
  k = FloatField('k', [validators.DataRequired()])
  v = FloatField('v', [])
  tf = FloatField('tf', [validators.DataRequired()])
  im = FloatField('im', [validators.DataRequired()])
  iv = FloatField('iv', [validators.DataRequired()])
  df = BooleanField('Displacement field', [])
  eigen = BooleanField('eigen', [])
  button = StringField('btn', [])
            
class Names():
  """
  Names/path of the files containing the simulations.
  """
  def __init__(self):
    self.eigenname, self.dfname, self.meshname = ['web/static/' + i for i in ('eig', 'df', 'mesh')]

  def addtime(self, t):
    self.eigenname += '?t=' + t + '.mp4'
    self.dfname += '?t=' + t + '.mp4'
    self.meshname += '?t=' + t + '.png'
    self.eigendir = self.eigenname[:-4]
    self.dfdir = self.dfname[:-4]
        
  def makestatic(self):
    self.eigenname = self.eigenname[11:]
    self.dfname = self.dfname[11:]
    self.meshname = self.meshname[11:]
    self.eigendir = self.eigenname[:-4]
    self.dfdir = self.dfname[:-4]

@app.route("/",  methods=['POST', 'GET'])
def mainpage():
  form = AppForm(request.form)
  names = Names()
  names.addtime(str(time.time())) # to have unique name
  check_boxes = {'df': False, 'eigen': False}
  open_page = {'open_page': 'mesh'}
  
  if request.method == 'POST':# and form.validate():
    if form.button.data == 'Redraw':
      app.backend.redraw(form.pi.data)
    else:
      app.backend.reset_constants(form.m.data, form.k.data, form.v.data, form.iv.data, form.im.data)
      open_page = {'open_page': 'physics'}
        
    if form.data['df']:                        
      app.backend.compute_df(form.data['tf'], names.dfname)
      check_boxes['df'] = True

    if form.data['eigen']:
      app.backend.plot_eigenva(names.eigenname[:-4])
      check_boxes['eigen'] = True

  app.backend.plot_mesh(names.meshname)
  names.makestatic()
  new_form = dict(app.backend.data, **check_boxes, **open_page)
  return render_template('app.html', form=new_form, names=names, numberparticles=app.backend.number_particles, numbereigframes=app.backend.number_eigenva)
  
if __name__ == "__main__":
  app.run(host='0.0.0.0', port=80)
